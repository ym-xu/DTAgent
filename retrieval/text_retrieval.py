import os
import json
from tqdm import tqdm
from ragatouille import RAGPretrainedModel

from retrieval.base_retrieval import BaseRetrieval
from mydatasets.base_dataset import BaseDataset

class ColbertRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config

    # create index for documents
    def prepare(self, dataset: BaseDataset):
        samples = dataset.load_data(use_retreival=True)
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        # build doc_id : index_path pairs
        doc_index:dict = {}
        error = 0
        for sample in tqdm(samples):
            # if already have indrx file, continue
            if self.config.r_text_index_key in sample and os.path.exists(sample[self.config.r_text_index_key]):
                continue
            # for other samples with same doc_id, we already have index
            if sample[self.config.doc_key] in doc_index:
                sample[self.config.r_text_index_key] = doc_index[sample[self.config.doc_key]]
                continue
            # first sample of this document
            content_list = dataset.load_processed_content(sample)
            text = [content.txt.replace("\n", "") for content in content_list]
            
            try:
                index_path = RAG.index(index_name=dataset.config.name+ "-" + self.config.text_question_key + "-" + sample[self.config.doc_key], collection=text)
                doc_index[sample[self.config.doc_key]] = index_path
                sample[self.config.r_text_index_key] = index_path
            except Exception as e:
                error += 1
                if error>len(samples)/100:
                    print("Too many error cases. Exit process.")
                    import sys
                    sys.exit(1)
                print(f"Error processing {sample[self.config.doc_key]}: {e}")
                sample[self.config.r_text_index_key] = ""
            
        dataset.dump_data(samples, use_retreival = True)
            
        return samples


    def find_top_k(self, dataset: BaseDataset, force_prepare=False):
        top_k = self.config.top_k
        samples = dataset.load_data(use_retreival=True)

        # if we don't have index file, generate them
        if self.config.r_text_index_key not in samples[0] or force_prepare:
            samples = self.prepare(dataset)