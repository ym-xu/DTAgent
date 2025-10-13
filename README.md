# image des generation in dict

python scripts/prep_dom_img.py 

add mineru:
python 

# build dom block

python scripts/build_blocks.py


# Build Tree

## adapte：

python -m src.adapter.adapter_v2 --in-file /path/to/doc_dir/content_list.json

python -m src.adapter.adapter_v2 --in-dir /path/to/doc_dir/

## enhance:

python -m src.enhancer.enhancer_2 --in-file /path/to/doc_dir/content_list.adapted.json

python -m src.enhancer.enhancer_2 --in-dir /path/to/doc_dir/

## build:

python -m src.builder.tree_builder --in-file /path/to/doc_dir/content_list.enhanced.json

python -m src.builder.tree_builder --in-dir /path/to/doc_dir/

# Prepare for Agentic DU

## index:

python -m src.index.build_index  --in-file ./../../../data/users/yiming/dtagent/MinerU_25_MMLB/2023.acl-long.386/doctree.mm.json --include-leaves