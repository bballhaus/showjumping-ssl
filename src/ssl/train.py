"""Joint SSL pretraining: InfoNCE on two augmented views + temporal-order
prediction on a shuffled triple. Logs losses to a CSV + saves encoder weights.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import SSLClipDataset, VenueBalancedSampler
from .encoder import R2Plus1DEncoder
from .grl import DomainHead, dann_lambda
from .losses import TemporalOrderHead, info_nce


def train(clips_dir: Path, out_dir: Path, epochs: int = 20, batch_size: int = 16,
          lr: float = 1e-3, tau: float = 0.1, lambda_order: float = 0.3,
          n_workers: int = 4, device: str = "cuda", kinetics_init: bool = False,
          lambda_domain: float = 0.0, balance_venues: bool = False) -> None:
    """Joint SSL pretraining.

    lambda_domain > 0 adds the domain-adversarial venue-suppression objective;
    balance_venues swaps the random shuffle for a venue-balanced sampler. Both
    target the milestone's venue-confound finding.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SSLClipDataset(clips_dir)
    if len(ds) == 0:
        raise SystemExit(f"[ssl/train] no clips found in {clips_dir}")
    sampler = VenueBalancedSampler(ds) if balance_venues else None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
                    num_workers=n_workers, pin_memory=True, drop_last=True)

    encoder = R2Plus1DEncoder(embed_dim=128, kinetics_init=kinetics_init).to(device)
    order_head = TemporalOrderHead(encoder.feat_dim, n_perms=6).to(device)
    use_domain = lambda_domain > 0 and ds.n_domains > 1
    domain_head = DomainHead(encoder.feat_dim, ds.n_domains).to(device) if use_domain else None

    params = list(encoder.parameters()) + list(order_head.parameters())
    if domain_head is not None:
        params += list(domain_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    total_steps = epochs * max(1, len(dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow(["epoch", "step", "loss_contrastive", "loss_order",
                         "loss_domain", "loss_total", "time_s"])

        t0 = time.time()
        gstep = 0
        for epoch in range(epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for step, batch in enumerate(dl):
                view_a = batch["view_a"].to(device, non_blocking=True)
                view_b = batch["view_b"].to(device, non_blocking=True)
                shuf_x = batch["shuffle_x"].to(device, non_blocking=True)
                shuf_y = batch["shuffle_y"].to(device, non_blocking=True)

                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    if domain_head is not None:
                        f_a = encoder.features(view_a)
                        z_a = nn.functional.normalize(encoder.projector(f_a), dim=-1)
                    else:
                        z_a = encoder(view_a)
                    z_b = encoder(view_b)
                    loss_c = info_nce(z_a, z_b, tau=tau)

                    f_s = encoder.features(shuf_x)
                    logits = order_head(f_s)
                    loss_o = nn.functional.cross_entropy(logits, shuf_y)

                    loss = loss_c + lambda_order * loss_o

                    loss_d = torch.zeros((), device=device)
                    if domain_head is not None:
                        dom_y = batch["domain_y"].to(device, non_blocking=True)
                        lambd = dann_lambda(gstep, total_steps)
                        dom_logits = domain_head(f_a, lambd=lambd)
                        loss_d = nn.functional.cross_entropy(dom_logits, dom_y)
                        loss = loss + lambda_domain * loss_d

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                gstep += 1

                writer.writerow([epoch, step, float(loss_c), float(loss_o),
                                 float(loss_d), float(loss), round(time.time() - t0, 2)])
                logf.flush()
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step}/{len(dl)} "
                          f"L_c={float(loss_c):.3f} L_o={float(loss_o):.3f} "
                          f"L_d={float(loss_d):.3f}")

            ckpt = {"encoder": encoder.state_dict(),
                    "order_head": order_head.state_dict(), "epoch": epoch}
            if domain_head is not None:
                ckpt["domain_head"] = domain_head.state_dict()
            torch.save(ckpt, out_dir / "encoder.pt")
    print(f"[ssl/train] done. log={log_path}, ckpt={out_dir/'encoder.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lambda-order", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--kinetics-init", action="store_true",
                    help="Initialize from Kinetics-400 weights (proposal fallback path).")
    ap.add_argument("--lambda-domain", type=float, default=0.0,
                    help="Weight on the domain-adversarial venue-suppression loss (0 = off).")
    ap.add_argument("--balance-venues", action="store_true",
                    help="Use a venue-balanced sampler so each source video is seen equally.")
    args = ap.parse_args()

    train(args.clips, args.out, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, tau=args.tau, lambda_order=args.lambda_order,
          n_workers=args.workers, device=args.device, kinetics_init=args.kinetics_init,
          lambda_domain=args.lambda_domain, balance_venues=args.balance_venues)


if __name__ == "__main__":
    main()

