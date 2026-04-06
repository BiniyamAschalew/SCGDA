import os.path as osp

from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree

from  data.data_loaders.airport import AirportDataset
from  data.data_loaders.blog import BlogDataset
from  data.data_loaders.citation import CitationDataset
from  data.data_loaders.twitch import TwitchDataset
from  data.data_loaders.mag import MAGDataset

from  utils.data_utils.svd_transform import svd_transform

def get_dataset(domain: str, config: dict):
    """load the dataset for the specific domain"""

    domain = domain.lower()

    domain_list = config["data"]["domains"]
    domain_name = {dom.lower(): dom for dom in domain_list}

    if domain not in domain_name:
        raise ValueError(
            f"Invalid domain {domain} for dataset {config['data']['name']}"
        )

    # convert to correct case
    domain = domain_name[domain]
    pre_transform = None

    if config["model"]["name"].lower() == "specreg":
        pre_transform = svd_transform
    
    if domain in {"BRAZIL", "EUROPE", "USA"}:
        dataset = AirportDataset(domain, config, pre_transform=pre_transform)

    elif domain in {"Blog1", "Blog2"}:
        dataset = BlogDataset(domain, config, pre_transform=pre_transform)

    elif domain in {"ACMv9", "Citationv1", "DBLPv7"}:
        dataset = CitationDataset(domain, config, pre_transform=pre_transform)

    elif domain in {"DE", "EN", "ES", "FR", "PT", "RU"}:
        dataset = TwitchDataset(domain, config, pre_transform=pre_transform)

    elif domain in {"MAG_CN", "MAG_DE", "MAG_FR", "MAG_RU", "MAG_JP", "MAG_US"}:
        dataset = MAGDataset(domain, config, pre_transform=pre_transform)

    else:
        raise ValueError(f"Invalid domain {domain}")

    return dataset


def get_max_degree(config):
    """get the maximum degree across all domains"""

    degree_list = []
    for domain in config["data"]["domains"]:
        dataset = get_dataset(domain, config)
        data = dataset[0]

        data_degree = max(degree(data.edge_index[0, :]))
        degree_list.append(data_degree)

    max_degree = int(max(degree_list))
    return max_degree


def build_dataset(config):

    source = config["expt"]["source"]
    target = config["expt"]["target"]

    source_dataset = get_dataset(source, config)
    target_dataset = get_dataset(target, config)

    # for airport dataset, we use one-hot degree as node features
    if config["data"]["name"] == "Airport":

        max_degree = get_max_degree(config)
        source_transform = OneHotDegree(max_degree)
        target_transform = OneHotDegree(max_degree)

        source_dataset.transform = source_transform
        target_dataset.transform = target_transform

    if config["expt"]["verbose"]:
        print(f"\n == Loaded datasets ==\n Source: {source}\n Target: {target}\n")

    return source_dataset, target_dataset


