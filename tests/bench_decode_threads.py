"""Benchmark A/B: decode com e sem CAP_PROP_N_THREADS.

Mede APENAS o decode (leitura de todos os frames), sem inferência.
Isso dá o teto do ganho possível: se nem o decode isolado ficar mais
rápido, a mudança no refacer.py não tem como ajudar e é descartada.
Se ficar, o ganho no job real ainda será menor (a inferência domina),
mas aí vale um segundo teste no pipeline completo.

Uso:
    python tests/bench_decode_threads.py caminho/do/video.mp4 [repeticoes]
"""
import sys
import time

import cv2


def decode_all(video_path, n_threads=None):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if n_threads is not None:
        ok = cap.set(cv2.CAP_PROP_N_THREADS, n_threads)
        if not ok:
            print(f"  [aviso] CAP_PROP_N_THREADS nao suportado neste build do OpenCV ({cv2.__version__})")
    start = time.perf_counter()
    frames = 0
    while True:
        flag, _ = cap.read()
        if not flag:
            break
        frames += 1
    elapsed = time.perf_counter() - start
    cap.release()
    return frames, elapsed


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    video_path = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"OpenCV {cv2.__version__} | video: {video_path} | {reps}x cada variante\n")

    results = {}
    for label, n_threads in [("baseline (default)", None), ("N_THREADS=0 (todos os cores)", 0)]:
        times = []
        for i in range(reps):
            frames, elapsed = decode_all(video_path, n_threads)
            times.append(elapsed)
            print(f"{label}: run {i + 1}/{reps} -> {frames} frames em {elapsed:.2f}s ({frames / elapsed:.1f} fps)")
        results[label] = min(times)

    base = results["baseline (default)"]
    threaded = results["N_THREADS=0 (todos os cores)"]
    delta = (base - threaded) / base * 100
    print(f"\nMelhor baseline: {base:.2f}s | melhor com threads: {threaded:.2f}s | ganho: {delta:+.1f}%")
    if delta < 5:
        print("Veredito: ganho dentro do ruido -> NAO adotar no refacer.py")
    else:
        print("Veredito: decode acelerou -> vale um segundo teste no job completo (reface de verdade)")


if __name__ == "__main__":
    main()
