from src.models import evaluate

# convenience entrypoint when run from the command line or imported
if __name__ == '__main__':
    df = evaluate.inspect_labels('dataset/temp_eval.csv')
    # further manual exploration can be done here if needed
