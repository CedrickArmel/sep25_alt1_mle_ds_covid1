"""Check W&B project access and list available runs."""

import wandb

def main():
    # Configuration
    entity = "yebouetc"
    project = "radiocovid"

    print(f"Checking access to project: {entity}/{project}")

    try:
        # Initialize API
        api = wandb.Api()
        print("✅ W&B API initialized successfully")

        # Try to access the project
        runs = api.runs(f"{entity}/{project}")
        print(f"✅ Access granted to project {entity}/{project}")

        # Count and list runs
        runs_list = list(runs)
        print(f"📊 Found {len(runs_list)} runs in the project")

        if runs_list:
            print("\n📋 Recent runs:")
            for i, run in enumerate(runs_list[:10]):  # Show first 10
                print(f"  {i+1}. {run.name} (ID: {run.id})")
                if hasattr(run, 'state'):
                    print(f"      State: {run.state}")
                if hasattr(run, 'summary'):
                    # Show some metrics if available
                    metrics = {k: v for k, v in run.summary.items() if not k.startswith('_')}
                    if metrics:
                        print(f"      Metrics: {list(metrics.keys())[:3]}...")  # Show first 3 metrics
                print()

        else:
            print("❌ No runs found in this project")

    except Exception as e:
        print(f"❌ Error accessing project: {e}")
        print("\nPossible issues:")
        print("- You don't have access to this project")
        print("- The project doesn't exist")
        print("- Your API key is invalid")
        print("- Network connectivity issues")

if __name__ == "__main__":
    main()