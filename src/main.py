from src.data_import import load_data, combine_data
from src.data_analysis import check_df, grab_col_names, missing_values_table
from src.data_visualization import plot_numerical_dist, plot_correlation_matrix, plot_categorical_dist
from src.data_preparation import prepare_data, one_hot_encoder
from src.data_modeling import split_data, evaluate_models

def main():
    # Load data
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    train_df, test_df = load_data(train_path, test_path)
    
    # Combine datasets
    df = combine_data(train_df, test_df)
    
    # Analyze data
    check_df(df)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    
    # Visualize data
    #plot_numerical_dist(df, num_cols)
    plot_correlation_matrix(df, num_cols)
    plot_categorical_dist(df, cat_cols)
    
    cat_cols = [col for col in cat_cols if col != 'loan_status']

    # Prepare data
    df = prepare_data(df, num_cols)
    df = one_hot_encoder(df, cat_cols)
    

    # Split data for modeling
    df1 = df[df['loan_status'].notnull()]
    y = df1["loan_status"]
    X = df1.drop(["loan_status"], axis=1)
    
    # Model and evaluate
    X_train, X_test, y_train, y_test, classifiers = split_data(X, y)
    evaluate_models(classifiers, X_test, y_test, X_train, y_train)

if __name__ == "__main__":
    main() 


