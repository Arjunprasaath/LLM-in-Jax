import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    script_dir = Path(__file__).resolve().parent
    jax_file = script_dir / "jax_metrics.json"
    torch_file = script_dir / "pytorch_metrics.json"

    if not jax_file.exists():
        print(f"File not found: {jax_file}. Run jax.py first!")
        return
    if not torch_file.exists():
        print(f"File not found: {torch_file}. Run pytorch.py first!")
        return

    with open(jax_file) as f:
        jax_data = json.load(f)
    with open(torch_file) as f:
        torch_data = json.load(f)

    # 1. Terminal Table
    print("\n" + "=" * 65)
    print(f"{'METRIC':<25} | {'JAX (Flax NNX)':<16} | {'PyTorch':<16}")
    print("=" * 65)

    jax_time = jax_data.get("total_time_sec", 0.0)
    torch_time = torch_data.get("total_time_sec", 0.0)
    print(f"{'Total Time':<25} | {jax_time:>13.2f} s | {torch_time:>13.2f} s")

    jax_steps = jax_data['step'][-1] if jax_data.get('step') else 0
    torch_steps = torch_data['step'][-1] if torch_data.get('step') else 0
    jax_speed = (jax_steps / jax_time) if jax_time > 0 else 0
    torch_speed = (torch_steps / torch_time) if torch_time > 0 else 0
    print(f"{'Throughput (steps/sec)':<25} | {jax_speed:>13.2f}   | {torch_speed:>13.2f}  ")

    jax_final_loss = jax_data['train_loss'][-1] if jax_data.get('train_loss') else float('nan')
    torch_final_loss = torch_data['train_loss'][-1] if torch_data.get('train_loss') else float('nan')
    print(f"{'Final Loss':<25} | {jax_final_loss:>13.4f}   | {torch_final_loss:>13.4f}  ")

    jax_final_ppl = jax_data['train_perplexity'][-1] if jax_data.get('train_perplexity') else float('nan')
    torch_final_ppl = torch_data['train_perplexity'][-1] if torch_data.get('train_perplexity') else float('nan')
    print(f"{'Final Perplexity':<25} | {jax_final_ppl:>13.2f}   | {torch_final_ppl:>13.2f}  ")
    print("=" * 65 + "\n")

    # 2. Overlaid Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(jax_data['step'], jax_data['train_loss'], label=f"JAX ({jax_time:.1f}s)", color='#E24A4A', linewidth=2)
    ax1.plot(torch_data['step'], torch_data['train_loss'], label=f"PyTorch ({torch_time:.1f}s)", color='#2B70C9', linewidth=2)
    ax1.set_title("Training Loss Comparison", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(jax_data['step'], jax_data['train_perplexity'], label="JAX", color='#E24A4A', linewidth=2)
    ax2.plot(torch_data['step'], torch_data['train_perplexity'], label="PyTorch", color='#2B70C9', linewidth=2)
    ax2.set_title("Training Perplexity Comparison", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Perplexity")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_img = script_dir / "jax_vs_pytorch.png"
    plt.savefig(out_img, dpi=150)
    print(f"Comparison plot saved to: {out_img}")

if __name__ == "__main__":
    main()
