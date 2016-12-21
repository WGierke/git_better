from preprocess import ColumnSumFilter, ColumnStdFilter, PolynomialTransformer
from training import find_best_repository_classification
from evaluation import get_cleaned_processed_df, drop_text_features
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    data_frame = get_cleaned_processed_df()
    data_frame = drop_text_features(data_frame)

    ppl = Pipeline([
        ('clmn_std_filter', ColumnStdFilter(min_std=10)),
        ('clmn_sum_filter', ColumnSumFilter(min_sum=10000)),
        ('poly_transf', PolynomialTransformer(degree=2))
    ])
    preprocessed_df = ppl.transform(data_frame)
    y_train = preprocessed_df.pop("label")
    find_best_repository_classification(preprocessed_df, y_train)