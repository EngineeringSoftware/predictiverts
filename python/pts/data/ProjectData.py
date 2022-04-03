from typing import *

from recordclass import RecordClass


class ProjectData(RecordClass):

    name: str = None
    url: str = None
    sha: str = None
    revision: str = None
    
    @classmethod
    def create(cls) -> "ProjectData":
        obj = ProjectData()
        return obj
