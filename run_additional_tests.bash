python index.py --dataset ia_contact --walk_bias Uniform --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset ia_contacts_hypertext_2009 --walk_bias Uniform --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset ia_enron_employees --walk_bias Uniform --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset ia_radoslaw_email --walk_bias Uniform --initial_edge_bias Uniform --weighted_node2vec

python index.py --dataset ia_contact --walk_bias Linear --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset ia_contacts_hypertext_2009 --walk_bias Linear --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset ia_enron_employees --walk_bias Linear --initial_edge_bias Uniform --weighted_node2vec
python index.py --dataset ia_radoslaw_email --walk_bias Linear --initial_edge_bias Uniform --weighted_node2vec

python index.py --dataset ia_contact --walk_bias Uniform --initial_edge_bias Linear --weighted_node2vec
python index.py --dataset ia_contacts_hypertext_2009 --walk_bias Uniform --initial_edge_bias Linear --weighted_node2vec
python index.py --dataset ia_enron_employees --walk_bias Uniform --initial_edge_bias Linear --weighted_node2vec
python index.py --dataset ia_radoslaw_email --walk_bias Uniform --initial_edge_bias Linear --weighted_node2vec

python index.py --dataset ia_contact --walk_bias Linear --initial_edge_bias Linear --weighted_node2vec
python index.py --dataset ia_contacts_hypertext_2009 --walk_bias Linear --initial_edge_bias Linear --weighted_node2vec
python index.py --dataset ia_enron_employees --walk_bias Linear --initial_edge_bias Linear --weighted_node2vec
python index.py --dataset ia_radoslaw_email --walk_bias Linear --initial_edge_bias Linear --weighted_node2vec



python index.py --dataset ia_contact --walk_bias Uniform --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_contacts_hypertext_2009 --walk_bias Uniform --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_enron_employees --walk_bias Uniform --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_radoslaw_email --walk_bias Uniform --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec

python index.py --dataset ia_contact --walk_bias Linear --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_contacts_hypertext_2009 --walk_bias Linear --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_enron_employees --walk_bias Linear --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_radoslaw_email --walk_bias Linear --initial_edge_bias Uniform --edge_operator hadamard --weighted_node2vec

python index.py --dataset ia_contact --walk_bias Uniform --initial_edge_bias Linear --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_contacts_hypertext_2009 --walk_bias Uniform --initial_edge_bias Linear --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_enron_employees --walk_bias Uniform --initial_edge_bias Linear --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_radoslaw_email --walk_bias Uniform --initial_edge_bias Linear --edge_operator hadamard --weighted_node2vec

python index.py --dataset ia_contact --walk_bias Linear --initial_edge_bias Linear --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_contacts_hypertext_2009 --walk_bias Linear --initial_edge_bias Linear --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_enron_employees --walk_bias Linear --initial_edge_bias Linear --edge_operator hadamard --weighted_node2vec
python index.py --dataset ia_radoslaw_email --walk_bias Linear --initial_edge_bias Linear --edge_operator hadamard --weighted_node2vec



