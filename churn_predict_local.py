import pandas as pd
import pickle
import json
import os

# controling user input, slightly!
while True:

    model_name = input('Pls specify file name of saved model: ').strip().lower().replace(" ", "")
    print(model_name)

    model_path = ''.join(os.path.dirname(os.path.abspath(__file__)) + '\\' + model_name + '.bin')

    if not os.path.exists(model_path):
        print('Missing file!, ensure correct file name')
    else:
        break

while True:
    threshold = float(input('Pls specify classification threshold from 0 to 1 (i.e.: 0.5): '))

    if (threshold < 0) or (threshold > 1):
        print('wrong input, threshold  must be between 0 and 1 (i.e.: 0.5)')
    else:
        break

while True:

    cus = input('Pls specify file name to load customers data: ').strip().lower().replace(" ", "")

    cust_path = ''.join(os.path.dirname(os.path.abspath(__file__)) + '\\' + cus + '.json')

    if not os.path.exists(cust_path):
        print('Missing file!, ensure correct file name')
    else:
        break

# Loading model

print('\n', '-' * 40, '\n', f'Loading model.....', '\n', sep = '')

with open(model_path, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
print('Done.', '\n' * 2, dv, '\n' * 2, model, '\n', '-' * 40, '\n', sep = '')

def gen_pred():

    # Loading customer file
    print(f'Loading customer file.....', '\n', sep = '')

    with open(cust_path, 'rb') as file:
        f_read = file.read()
        customers = json.loads(f_read)

    print('Done', '\n', '-' * 40, '\n', sep = '')

    # generate predictions
    print('Generating predictions.....', '\n', sep = '')

    results = {}

    for cust in customers.items():
        # transform
        x = dv.transform(cust[1])
        p_prob = model.predict_proba(x)[:, 1][0]
        churn = (p_prob >= threshold)

        # update results
        results.update({f'{cust[0]}': {f'churn proba': round(p_prob, 2), f'chrun': churn}})
    
    print('Done', '\n', '-' * 40, '\n', sep = '')

    while True:

        usr_pref = input('Use \'df\' to return Dataframe or \'d\' to display results: ').strip().lower().replace(" ", "")

        if usr_pref == 'df':
            # return results as a dataframe
            res_df = pd.DataFrame.from_dict(results,'index').rename_axis('customer').reset_index()
            res_df.to_csv('res_df.csv', index = False)
            path = ''.join(os.path.dirname(os.path.abspath(__file__)) + '\\' + 'res_df.csv')
            print(f'Results saved to {path} ', '\n', '-' * 40, '\n', sep = '')
            break

        elif usr_pref == 'd':
            # display results
            print('\n', '-' * 40, '\n', results, '\n', '-' * 40, '\n', sep = '')
            break

        else:
            print('Wrong input! Either use \'df\' or \'d\' ')


if __name__ == '__main__':
    gen_pred()
