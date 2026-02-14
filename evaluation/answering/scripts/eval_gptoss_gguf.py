import argparse
import pandas as pd

from core import read_repo_ids, build_specs_from_list, filter_specs, run_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_file", default="./Endo_Test.csv")
    ap.add_argument("--models_list", default="./models_list.txt")
    ap.add_argument("--out_dir", default="./harrison_gptoss_gguf")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--llama_bin", default="/llama.cpp/build/bin/llama-server")
    ap.add_argument("--port_base", type=int, default=8080)
    args = ap.parse_args()

    df = pd.read_csv(args.eval_file)

    repo_ids = read_repo_ids(args.models_list)
    
    specs_all = build_specs_from_list(
        repo_ids, 
        llama_bin=args.llama_bin, 
        port_base=args.port_base
    )
    
    specs = filter_specs(specs_all, family="gptoss", kind="gguf")

    system_text = "You are an expert endocrinologist. Reply with exactly: Final answer: <A|B|C|D>"
    run_eval(df, specs, out_dir=args.out_dir, resume=args.resume,
             load_in_4bit=False, max_new_tokens=args.max_new_tokens,
             system_text=system_text)

if __name__ == "__main__":
    main()