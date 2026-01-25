from waitress import serve
import multiprocessing
from application import app

if __name__ == "__main__":
    num_cpus = multiprocessing.cpu_count()
    threads_per_worker = max(1, num_cpus)

    print("threads:", threads_per_worker)
    print("server started")

    serve(
        app,
        host="0.0.0.0",
        port=5000,
        threads=threads_per_worker
    )
