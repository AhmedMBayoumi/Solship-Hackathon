import modal

app = modal.App("zc-hackathon-test")

image = modal.Image.debian_slim().pip_install("matplotlib", "numpy")

@app.function(image=image, gpu="T4")
def test_gpu_and_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import subprocess

    gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()

    x = np.linspace(0, 10, 100)
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.plot(x, np.cos(x), label="cos(x)")
    ax.set_title(f"GPU Test OK\n{gpu_info}")
    ax.legend()
    fig.savefig("/tmp/test_plot.png", dpi=100, bbox_inches="tight")

    with open("/tmp/test_plot.png", "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    print("Running test on Modal GPU...")
    img_bytes = test_gpu_and_plot.remote()
    out_path = "C:/Ahmed Bayoumi/University/ZC Hackathon/test_plot.png"
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    print(f"Plot saved to {out_path}")
    print(f"Image size: {len(img_bytes)} bytes — success!")
