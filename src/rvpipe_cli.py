# src/rvpipe_cli.py
import typer
from pipeline.index import index_main
from pipeline.build import build_shards
from pipeline.train import train_main
from pipeline.export import export_onnx
from pipeline.serve import serve_main
from pipeline.evaluate import evaluate_main
from pipeline.bench import bench_main
import uvicorn

app = typer.Typer(no_args_is_help=True)

@app.command()
def index(config: str = "configs/index.yaml"):
    index_main(config)

@app.command()
def build(config: str = "configs/build.yaml"):
    build_shards(config)

@app.command()
def train(config: str = "configs/train.yaml"):
    train_main(config)

@app.command()
def export(config: str = "configs/export.yaml"):
    export_onnx(config)

@app.command()
def serve(config: str = "configs/serve.yaml"):
    fastapi_app = serve_main(config)
    uvicorn.run(fastapi_app,
                host="0.0.0.0",
                port=8000,
                workers=1)

@app.command()
def evaluate(config: str = "configs/eval.yaml"):
    evaluate_main(config)
    
@app.command()
def bench(config: str = "configs/bench.yaml"):
    bench_main(config)

if __name__ == "__main__":
    app()
