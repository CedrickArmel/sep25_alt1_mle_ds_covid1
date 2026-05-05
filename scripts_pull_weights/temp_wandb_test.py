import wandb

api = wandb.Api()
entity = 'yebouetc'
project = 'radiocovid'
query = {'name': 'skilled-haze-6'}
runs = api.runs(f'{entity}/{project}', query)
print('runs count', len(list(runs)))
for run in runs:
    print(run.name, run.id)
