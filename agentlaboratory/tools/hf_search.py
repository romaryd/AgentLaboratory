import numpy as np
from datasets import load_dataset, load_dataset_builder
from psutil._common import bytes2human
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

class HFDataSearch:
    def __init__(self, like_thr=3, dwn_thr=50) -> None:
        """
        Class for finding relevant huggingface datasets
        :param like_thr:
        :param dwn_thr:
        """
        self.dwn_thr = dwn_thr
        self.like_thr = like_thr
        self.ds = load_dataset("nkasmanoff/huggingface-datasets")["train"]

        # Initialize lists to collect filtered data
        filtered_indices = []
        filtered_descriptions = []
        filtered_likes = []
        filtered_downloads = []

        # Iterate over the dataset and filter based on criteria
        for idx, item in enumerate(self.ds):
            # Get likes and downloads, handling None values
            likes = int(item['likes']) if item['likes'] is not None else 0
            downloads = int(item['downloads']) if item['downloads'] is not None else 0

            # Check if likes and downloads meet the thresholds
            if likes >= self.like_thr and downloads >= self.dwn_thr:
                # Check if the description is a non-empty string
                description = item['description']
                if isinstance(description, str) and description.strip():
                    # Collect the data
                    filtered_indices.append(idx)
                    filtered_descriptions.append(description)
                    filtered_likes.append(likes)
                    filtered_downloads.append(downloads)

        # Check if any datasets meet all criteria
        if not filtered_indices:
            print("No datasets meet the specified criteria.")
            self.ds = []
            self.descriptions = []
            self.likes_norm = []
            self.downloads_norm = []
            self.description_vectors = None
            return  # Exit the constructor

        # Filter the datasets using the collected indices
        self.ds = self.ds.select(filtered_indices)

        # Update descriptions, likes, and downloads
        self.descriptions = filtered_descriptions
        self.likes = np.array(filtered_likes)
        self.downloads = np.array(filtered_downloads)

        # Normalize likes and downloads
        self.likes_norm = self._normalize(self.likes)
        self.downloads_norm = self._normalize(self.downloads)

        # Vectorize the descriptions
        self.vectorizer = TfidfVectorizer()
        self.description_vectors = self.vectorizer.fit_transform(self.descriptions)

    def _normalize(self, arr):
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr, dtype=float)
        return (arr - min_val) / (max_val - min_val)

    def retrieve_ds(self, query, N=10, sim_w=1.0, like_w=0.0, dwn_w=0.0):
        """
        Retrieves the top N datasets matching the query, weighted by likes and downloads.
        :param query: The search query string.
        :param N: The number of results to return.
        :param sim_w: Weight for cosine similarity.
        :param like_w: Weight for likes.
        :param dwn_w: Weight for downloads.
        :return: List of top N dataset items.
        """
        if not self.ds or self.description_vectors is None:
            print("No datasets available to search.")
            return []

        query_vector = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vector, self.description_vectors).flatten()
        # Normalize cosine similarities
        cosine_similarities_norm = self._normalize(cosine_similarities)
        # Compute final scores
        final_scores = (
                sim_w * cosine_similarities_norm +
                like_w * self.likes_norm +
                dwn_w * self.downloads_norm
        )
        # Get top N indices
        top_indices = final_scores.argsort()[-N:][::-1]
        # Convert indices to Python ints
        top_indices = [int(i) for i in top_indices]
        top_datasets = [self.ds[i] for i in top_indices]
        # check if dataset has a test & train set
        has_test_set = list()
        has_train_set = list()
        ds_size_info = list()
        for i in top_indices:
            try:
                dbuilder = load_dataset_builder(self.ds[i]["id"], trust_remote_code=True).info
            except Exception as e:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue

            if dbuilder.splits is None:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue
            # Print number of examples for
            has_test, has_train = "test" in dbuilder.splits, "train" in dbuilder.splits
            has_test_set.append(has_test)
            has_train_set.append(has_train)
            test_dwn_size, test_elem_size = None, None
            train_dwn_size, train_elem_size = None, None
            if has_test:
                test_dwn_size = bytes2human(dbuilder.splits["test"].num_bytes)
                test_elem_size = dbuilder.splits["test"].num_examples
            if has_train:
                train_dwn_size = bytes2human(dbuilder.splits["train"].num_bytes)
                train_elem_size = dbuilder.splits["train"].num_examples
            ds_size_info.append((test_dwn_size, test_elem_size, train_dwn_size, train_elem_size))
        for _i in range(len(top_datasets)):
            top_datasets[_i]["has_test_set"] = has_test_set[_i]
            top_datasets[_i]["has_train_set"] = has_train_set[_i]
            top_datasets[_i]["test_download_size"] = ds_size_info[_i][0]
            top_datasets[_i]["test_element_size"] = ds_size_info[_i][1]
            top_datasets[_i]["train_download_size"] = ds_size_info[_i][2]
            top_datasets[_i]["train_element_size"] = ds_size_info[_i][3]
        return top_datasets

    def results_str(self, results):
        """
        Provide results as list of results in human-readable format.
        :param results: (list(dict)) list of results from search
        :return: (list(str)) list of results in human-readable format
        """
        result_strs = list()
        for result in results:
            res_str = f"Dataset ID: {result['id']}\n"
            res_str += f"Description: {result['description']}\n"
            res_str += f"Likes: {result['likes']}\n"
            res_str += f"Downloads: {result['downloads']}\n"
            res_str += f"Has Testing Set: {result['has_test_set']}\n"
            res_str += f"Has Training Set: {result['has_train_set']}\n"
            res_str += f"Test Download Size: {result['test_download_size']}\n"
            res_str += f"Test Dataset Size: {result['test_element_size']}\n"
            res_str += f"Train Download Size: {result['train_download_size']}\n"
            res_str += f"Train Dataset Size: {result['train_element_size']}\n"
            result_strs.append(res_str)
        return result_strs
