python index.py --dataset fb_forum --walk_bias Exponential --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset ia_contact --walk_bias Exponential --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset ia_contacts_hypertext_2009 --walk_bias Exponential --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset ia_email_eu --walk_bias Exponential --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset ia_enron_employees --walk_bias Exponential --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset ia_radoslaw_email --walk_bias Exponential --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset soc_sign_bitcoin_alpha --walk_bias Exponential --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset wiki_elections --walk_bias Exponential --initial_edge_bias Uniform --weighted_node2vec

python index.py --dataset fb_forum --walk_bias Exponential --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_contact --walk_bias Exponential --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_contacts_hypertext_2009 --walk_bias Exponential --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_email_eu --walk_bias Exponential --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_enron_employees --walk_bias Exponential --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_radoslaw_email --walk_bias Exponential --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset soc_sign_bitcoin_alpha --walk_bias Exponential --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset wiki_elections --walk_bias Exponential --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
