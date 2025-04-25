#!/usr/bin/env python3
# encoding: utf-8

from frame import Frame

import os
import uuid
import rospy
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class DatasetEntry:
    id: str
    frame: Frame
    corners: np.ndarray


class Dataset:
    def __init__(self, entries: List[DatasetEntry]):
        self.entries = entries

    def save(self, path: os.PathLike):
        data_dict = {"ids": []}
        for entry in self.entries:
            eid = entry.id
            data_dict["ids"].append(eid)
            data_dict[f"{eid}_image"] = entry.frame.image
            data_dict[f"{eid}_timestamp"] = entry.frame.timestamp.to_sec()
            data_dict[f"{eid}_corners"] = entry.corners
        np.savez_compressed(path, **data_dict)

    @classmethod
    def load(cls, path: os.PathLike) -> 'Dataset':
        data = np.load(path, allow_pickle=True)
        entries = []
        ids = data["ids"]
        for eid in ids:
            image = data[f"{eid}_image"]
            timestamp = rospy.Time.from_sec(float(data[f"{eid}_timestamp"]))
            corners = data[f"{eid}_corners"]
            entry = DatasetEntry(
                id=eid,
                frame=Frame(image=image, timestamp=timestamp),
                corners=corners
            )
            entries.append(entry)
        return cls(entries)

    @staticmethod
    def create_entry(image: np.ndarray, timestamp: rospy.Time, corners: np.ndarray) -> DatasetEntry:
        eid = str(uuid.uuid4())
        return DatasetEntry(id=eid, frame=Frame(image=image, timestamp=timestamp), corners=corners)
