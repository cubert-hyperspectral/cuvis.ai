class Metadata:

    C_ATTRIB_LIST = ["shape", "wavelengths_nm", "references", "bit_depth", "integration_time_us", "framerate", "flags"]
    
    def __init__(self, name:str, fileset_metadata: dict={}):
        self.name:str = name
        self.shape:tuple = ()
        self.wavelengths_nm:list[int] = []
        self.bit_depth:int = None
        self.references:dict = {}
        self.integration_time_us:float = None
        self.framerate:float = None
        self.flags:dict = {}
        if fileset_metadata:
            self._init_from_file(fileset_metadata)

    def _init_from_file(self, fileset_metadata:dict):
        meta = None
        try:
            if any(((filename in self.name) for filename in fileset_metadata["fileset"]["paths"])):
                meta = fileset_metadata["fileset"]
        except KeyError:
            try:
                meta = fileset_metadata["fileset"]
            except KeyError:
                meta = fileset_metadata
        if meta is not None:
            for attrib_name in self.C_ATTRIB_LIST:
                try:
                    setattr(self, attrib_name, meta[attrib_name])
                except:
                    pass
    
    def __str__(self) -> str:
        out_str:str = F"name: {self.name}"
        for attrib_name in self.C_ATTRIB_LIST:
            out_str += F", {attrib_name}: {getattr(self, attrib_name)}"
        return out_str

    def __repr__(self) -> str:
        return F"{{{self.__str__()}}}";
