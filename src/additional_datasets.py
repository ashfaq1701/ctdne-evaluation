from io import StringIO
import pandas as pd
import numpy as np
import os
from stellargraph import StellarGraph


def resolve_path(*possible_paths):
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the specified paths exist: {possible_paths}")


class IAContact:
    def load(self):
        edges_path = resolve_path(
            'data/ia-contact.edges',
            '../data/ia-contact.edges'
        )
        edges = pd.read_csv(
            edges_path,
            sep=r'\s+',
            header=None,
            names=["source", "target", "x", "time"],
            usecols=["source", "target", "time"],
            engine='python'
        )

        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


class IARadoslawEmail:
    def load(self):
        edges_path = resolve_path(
            'data/ia-radoslaw-email.edges',
            '../data/ia-radoslaw-email.edges'
        )
        edges = pd.read_csv(
            edges_path,
            sep=r'\s+',
            header=None,
            names=["source", "target", "x", "time"],
            usecols=["source", "target", "time"],
            engine='python',
            skiprows=2
        )

        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


class IAContactsHypertext2009:
    def load(self):
        edges_path = resolve_path(
            'data/ia-contacts_hypertext2009.edges',
            '../data/ia-contacts_hypertext2009.edges'
        )
        edges = pd.read_csv(
            edges_path,
            sep=r',',
            header=None,
            names=["source", "target", "time"],
            usecols=["source", "target", "time"],
            engine='python'
        )

        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


class FBForum:
    def load(self):
        edges_path = resolve_path(
            'data/fb-forum.edges',
            '../data/fb-forum.edges'
        )
        edges = pd.read_csv(
            edges_path,
            sep=r',',
            header=None,
            names=["source", "target", "time"],
            usecols=["source", "target", "time"],
            engine='python'
        )

        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


class SocSignBitcoinAlpha:
    def load(self):
        edges_path = resolve_path(
            'data/out.soc-sign-bitcoinalpha',
            '../data/out.soc-sign-bitcoinalpha'
        )
        edges = pd.read_csv(
            edges_path,
            sep=r'\s+',
            header=None,
            skiprows=1,
            names=["source", "target", "x", "time"],
            usecols=["source", "target", "time"],
            engine='python'
        )

        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


class IAEmailEU:
    def load(self):
        edges_path = resolve_path(
            'data/email-Eu-core-temporal.txt',
            '../data/email-Eu-core-temporal.txt'
        )
        edges = pd.read_csv(
            edges_path,
            sep=r'\s+',
            header=None,
            names=["source", "target", "time"],
            usecols=["source", "target", "time"],
            engine='python'
        )

        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


class WikiElections:
    def load(self):
        edges_path = resolve_path(
            'data/soc-wiki-elec.edges',
            '../data/soc-wiki-elec.edges'
        )
        edges = pd.read_csv(
            edges_path,
            sep=r'\s+',
            header=None,
            names=["source", "target", "x", "time"],
            usecols=["source", "target", "time"],
            engine='python'
        )

        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


class AlibabaData:
    def load(self):
        edges_path = resolve_path(
            'data/data_alibaba.parquet',
            '../data/data_alibaba.parquet'
        )
        edges = pd.read_parquet(edges_path)
        edges.columns = ["source", "target", "time"]

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges

class FBLinks:
    def load(self):
        edges_path = resolve_path(
            'data/out.facebook-wosn-links',
            '../data/out.facebook-wosn-links'
        )
        edges = pd.read_csv(
            edges_path,
            sep=r'\s+',
            header=None,
            names=["source", "target", "x", "time"],
            usecols=["source", "target", "time"],
            engine='python',
            skiprows=2
        )

        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges
