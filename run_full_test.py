import pandas as pd
from autorl_framework.simulation.auto_rl_engine import AutoRLEngine
import os

def prepare_data(df_control, df_test):
    new_cols = {
        'Campaign Name': 'campaign_name', 'Date': 'date', 'Spend [USD]': 'spend',
        '# of Impressions': 'impressions', 'Reach': 'reach', '# of Website Clicks': 'website_clicks',
        '# of Searches': 'searches', '# of View Content': 'view_content',
        '# of Add to Cart': 'add_to_cart', '# of Purchase': 'purchases'
    }
    df_control = df_control.rename(columns=new_cols)
    df_test = df_test.rename(columns=new_cols)
    for col in df_control.columns:
        if df_control[col].dtype in ['float64', 'int64'] and df_control[col].isnull().any():
            mean_val = df_control[col].mean()
            df_control[col] = df_control[col].fillna(mean_val)
    for col in df_test.columns:
        if df_test[col].dtype in ['float64', 'int64'] and df_test[col].isnull().any():
            mean_val = df_test[col].mean()
            df_test[col] = df_test[col].fillna(mean_val)
    df_full = pd.concat([df_control, df_test], ignore_index=True)
    df_full['date'] = pd.to_datetime(df_full['date'], format='%d.%m.%Y', errors='coerce')
    df_full = df_full.sort_values('date')
    df_full['binary_conversion'] = (df_full['purchases'] > 0).astype(int)
    df_full['purchases_reward'] = df_full['purchases']
    return df_full

def main():
    print("Iniciando teste automático de todos os algoritmos e otimizadores...")
    control_path = os.path.join('data', 'control_group.csv')
    test_path = os.path.join('data', 'test_group.csv')
    df_control = pd.read_csv(control_path, sep=';')
    df_test = pd.read_csv(test_path, sep=';')
    df_full = prepare_data(df_control, df_test)
    engine = AutoRLEngine(data=df_full, arms_col='campaign_name', reward_col='purchases_reward')
    results, log_file = engine.run_full_test_and_save(n_iterations=3)
    print(f"\nTeste automático concluído! Log salvo em: {log_file}")

if __name__ == "__main__":
    main() 