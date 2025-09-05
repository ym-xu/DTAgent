import os
import json
from tqdm import tqdm
from ragatouille import RAGPretrainedModel

from retrieval.base_retrieval import BaseRetrieval
from mydatasets.base_dataset import BaseDataset

class BlockRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config

    def _extract_block_text(self, block):
        """
        从block中提取用于检索的文本
        
        Args:
            block: 单个block数据
            
        Returns:
            str: 合并的文本内容
        """
        # 提取block标题和内容
        title = block.get('title', '')
        parent_path = ' > '.join(block.get('parent_path', []))
        
        # 从elements中提取文本和描述
        text_parts = []
        if title:
            text_parts.append(f"Title: {title}")
        if parent_path:
            text_parts.append(f"Context: {parent_path}")
            
        for element in block.get('elements', []):
            # 添加文本内容
            if element.get('text'):
                text_parts.append(element['text'])
            # 添加图像/表格描述
            if element.get('description'):
                text_parts.append(f"Description: {element['description']}")
                
        return ' '.join(text_parts).replace('\n', ' ').strip()

    def _load_document_blocks(self, blocks_file_path):
        """
        加载单个文档的blocks并建立映射
        
        Args:
            blocks_file_path: blocks文件路径
            
        Returns:
            tuple: (blocks列表, block_id到索引的映射字典)
        """
        if not os.path.exists(blocks_file_path):
            return [], {}
            
        with open(blocks_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        blocks = data.get('blocks', [])
        
        # 建立 block_id 到索引的映射
        block_id_to_index = {block['block_id']: idx for idx, block in enumerate(blocks)}
        
        return blocks, block_id_to_index

    def prepare(self, dataset: BaseDataset):
        """
        为所有文档的blocks构建ColBERT检索索引
        
        Args:
            dataset: BaseDataset实例
            
        Returns:
            list: 更新后的samples
        """
        samples = dataset.load_data(use_retreival=True)
        
        # 测试模式：只处理前10个样本
        print(f"Debug - Total samples: {len(samples)}")
        samples = samples[:10]
        print(f"Debug - Testing with {len(samples)} samples only")
        
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        
        # build doc_id : index_path pairs
        doc_index: dict = {}
        error = 0
        
        for sample in tqdm(samples, desc="Building block indices"):
            # if already have index file, continue (but check if mapping file exists too)
            if (self.config.r_block_index_key in sample and 
                os.path.exists(sample[self.config.r_block_index_key]) and
                os.path.exists(os.path.join(sample[self.config.r_block_index_key], 'block_id_map.json'))):
                continue
                
            # for other samples with same doc_id, we already have index
            if sample[self.config.doc_key] in doc_index:
                sample[self.config.r_block_index_key] = doc_index[sample[self.config.doc_key]]
                continue
                
            # first sample of this document - load blocks
            doc_name = dataset.EXTRACT_DOCUMENT_ID(sample)
            blocks_file = os.path.join(
                dataset.config.base_dir if hasattr(dataset.config, 'base_dir') else '/tmp',
                'tmp', 
                dataset.config.name, 
                'blocks', 
                f"{doc_name}_blocks.json"
            )
            
            blocks, block_id_mapping = self._load_document_blocks(blocks_file)
            if not blocks:
                print(f"No blocks found for {doc_name} at {blocks_file}")
                sample[self.config.r_block_index_key] = ""
                continue
                
            # Extract text from blocks and filter out empty ones
            valid_blocks = []
            valid_block_texts = []
            
            for block in blocks:
                text = self._extract_block_text(block)
                if text and text.strip():
                    valid_blocks.append(block)
                    valid_block_texts.append(text)
            
            if not valid_blocks:
                print(f"No valid blocks found for {doc_name}")
                sample[self.config.r_block_index_key] = ""
                continue
            
            print(f"Debug - Doc: {doc_name}")
            print(f"Debug - Original blocks count: {len(blocks)}")
            print(f"Debug - Valid blocks count: {len(valid_blocks)}")
            print(f"Debug - Valid block IDs: {[b['block_id'] for b in valid_blocks[:5]]}...")
            
            try:
                index_name = f"{dataset.config.name}-block-{self.config.block_question_key}-{sample[self.config.doc_key]}"
                index_path = RAG.index(index_name=index_name, collection=valid_block_texts)
                
                # Store block_id mapping for later retrieval (based on valid_blocks only)
                mapping_path = os.path.join(index_path, 'block_id_map.json')
                # Create reverse mapping: passage_id -> block_id (using valid_blocks)
                passage_id_to_block_id = {str(idx): block['block_id'] for idx, block in enumerate(valid_blocks)}
                
                with open(mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(passage_id_to_block_id, f, indent=2, ensure_ascii=False)
                
                doc_index[sample[self.config.doc_key]] = index_path
                sample[self.config.r_block_index_key] = index_path
                
                print(f"Created index for {doc_name}: {len(valid_blocks)} valid blocks (from {len(blocks)} total)")
                
            except Exception as e:
                error += 1
                if error > len(samples) / 100:
                    print("Too many error cases. Exit process.")
                    import sys
                    sys.exit(1)
                print(f"Error processing {sample[self.config.doc_key]}: {e}")
                sample[self.config.r_block_index_key] = ""
                
        dataset.dump_data(samples, use_retreival=True)
        return samples

    def find_sample_top_k(self, sample, top_k: int, block_id_key: str = None):
        """
        为单个sample找到top-k相关的blocks
        
        Args:
            sample: 数据样本
            top_k: 返回的block数量
            block_id_key: 如果sample中有指定的block范围，用此key获取
            
        Returns:
            tuple: (top_block_ids, top_block_scores)
        """
        if not sample.get(self.config.r_block_index_key) or not os.path.exists(sample[self.config.r_block_index_key]):
            print(f"Index not found for {sample.get(self.config.r_block_index_key, 'None')}")
            return [], []
        
        print(f"Debug - Using index path: {sample[self.config.r_block_index_key]}")
            
        # Load block_id mapping
        mapping_path = os.path.join(sample[self.config.r_block_index_key], 'block_id_map.json')
        if not os.path.exists(mapping_path):
            print(f"Block mapping not found: {mapping_path}")
            return [], []
        
        print(f"Debug - Using mapping path: {mapping_path}")
            
        with open(mapping_path, 'r', encoding='utf-8') as f:
            passage_id_to_block_id = json.load(f)
            
        query = sample[self.config.block_question_key]
        
        # Search using RAG
        RAG = RAGPretrainedModel.from_index(sample[self.config.r_block_index_key])
        
        # Check actual number of passages in the index
        pid_docid_path = os.path.join(sample[self.config.r_block_index_key], 'pid_docid_map.json')
        if os.path.exists(pid_docid_path):
            with open(pid_docid_path, 'r') as f:
                pid_docid_data = json.load(f)
            actual_passages_count = len(pid_docid_data)
            print(f"Debug - Mapping entries: {len(passage_id_to_block_id)}, Actual passages: {actual_passages_count}")
        else:
            actual_passages_count = len(passage_id_to_block_id)
        
        results = RAG.search(query, k=min(actual_passages_count, 50))  # 限制最多50个结果
        
        # Convert passage_ids to block_ids  
        # Note: Multiple passages may belong to the same block
        block_scores = {}  # block_id -> best_score
        skipped_count = 0
        
        for result in results:
            pid = str(result['passage_id'])
            if pid in passage_id_to_block_id:
                block_id = passage_id_to_block_id[pid]
                score = result['score']
                
                # Keep the best score for each block
                if block_id not in block_scores or score > block_scores[block_id]:
                    block_scores[block_id] = score
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"Debug - Skipped {skipped_count} unmappable passages out of {len(results)}")
        
        # Sort blocks by score and take top_k
        sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_block_ids = [block_id for block_id, score in sorted_blocks]
        top_block_scores = [score for block_id, score in sorted_blocks]
        
        # Filter by block_id_key if provided
        if block_id_key and block_id_key in sample:
            allowed_block_ids = set(sample[block_id_key])
            filtered_ids = []
            filtered_scores = []
            
            for block_id, score in zip(top_block_ids, top_block_scores):
                if block_id in allowed_block_ids:
                    filtered_ids.append(block_id)
                    filtered_scores.append(score)
                    
            return filtered_ids[:top_k], filtered_scores[:top_k]
        
        return top_block_ids[:top_k], top_block_scores[:top_k]

    def find_top_k(self, dataset: BaseDataset, force_prepare=False):
        """
        为所有samples找到top-k相关的blocks
        
        Args:
            dataset: BaseDataset实例
            force_prepare: 是否强制重新构建索引
        """
        top_k = self.config.top_k
        samples = dataset.load_data(use_retreival=True)
        
        # 测试模式：只处理前10个样本
        print(f"Debug - Total samples for retrieval: {len(samples)}")
        samples = samples[:10]
        print(f"Debug - Testing retrieval with {len(samples)} samples only")

        # Force clear all existing index paths and rebuild (testing mode)
        print("Debug - Force clearing all index paths for clean rebuild...")
        cleared_count = 0
        for sample in samples:
            if self.config.r_block_index_key in sample:
                del sample[self.config.r_block_index_key]
                cleared_count += 1
        
        print(f"Debug - Cleared {cleared_count} existing index paths")
        print("Debug - Starting fresh prepare...")
        samples = self.prepare(dataset)

        for sample in tqdm(samples, desc="Finding top-k blocks"):
            top_block_ids, top_block_scores = self.find_sample_top_k(
                sample, 
                top_k=top_k, 
                block_id_key=getattr(dataset.config, 'block_id_key', None)
            )
            
            sample[self.config.r_block_key] = top_block_ids
            sample[self.config.r_block_key + "_score"] = top_block_scores
            
        path = dataset.dump_data(samples, use_retreival=True)
        print(f"Save block retrieval results at {path}.")