def get_paired_road_dataset():
    from .dataset import PairedRoadDataset
    return PairedRoadDataset

def get_synthesizer():
    from .synthetic_generator import RoadAdverseWeatherSynthesizer
    return RoadAdverseWeatherSynthesizer
