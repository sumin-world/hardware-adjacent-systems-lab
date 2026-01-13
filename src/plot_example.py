import skrf as rf
import matplotlib.pyplot as plt

def main():
    print("=== Channel Analyzer: Example dataset ===\n")

    # Example 2-port network bundled with scikit-rf
    ntwk = rf.data.ring_slot

    f_ghz = ntwk.f / 1e9
    s21_db = ntwk.s_db[:, 1, 0]  # S21: forward transmission
    s11_db = ntwk.s_db[:, 0, 0]  # S11: input reflection

    print("Lod example network")
    print(f"Frequency range: {f_ghz[0]:.2f} ~ {f_ghz[-1]:.2f} GHz")
    print(f"Points: {len(f_ghz)}\n")

    print("=== Metrics ===")
    print(f"S21 min  : {s21_db.min():.2f} dB (worst loss)")
    print(f"S21 mean : {s21_db.mean():.2f} dB")
    print(f"S11 mean : {s11_db.mean():.2f} dB (reflection)\n")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(f_ghz, s21_db, linewidth=2, label="S21")
    ax1.axhline(y=-3, linestyle="--", linewidth=1.5, label="-3 dB")
    ax1.set_title("Transmission (S21)")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(f_ghz, s11_db, linewidth=2, label="S11")
    ax2.set_title("Reflection (S11)")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.grid(True)
    ax2.legend()

    out_path = "outputs/result.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

if __name__ == "__main__":
    main()

