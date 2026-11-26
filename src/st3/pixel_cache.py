"""Optional immutable CPU FP32 processor outputs; no model/target caching."""
import ctypes
import os
import time

import torch


def available_ram():
    if os.name=='nt':
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','total_page','avail_page','total_virtual','avail_virtual','extended')]
        value=Memory();value.length=ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):raise RuntimeError('Cannot inspect free RAM')
        return value.available
    with open('/proc/meminfo') as handle:
        fields=dict(line.split(':',1) for line in handle)
    return int(fields['MemAvailable'].split()[0])*1024


class RamPixels:
    def __init__(self,reader,names,*,max_bytes=2*1024**3,margin_bytes=1024**3):
        self.reader=reader;self.names=tuple(sorted(set(names)));self.index={n:i for i,n in enumerate(self.names)}
        self.profile=False;self.reset_stats();start=time.perf_counter()
        required=len(self.names)*3*224*224*4;self.bytes=required
        # Reserve a decode/processor scratch allowance as well as the cache.
        reserve=margin_bytes+64*1024**2
        if not self.names or required>max_bytes or available_ram()<required+reserve:
            raise ValueError(f'CPU pixel cache needs {required} bytes plus {reserve} reserve; insufficient RAM or cache cap. Use --pixel-cache off or a larger-memory runtime; no silent fallback.')
        values=torch.empty((len(self.names),3,224,224),dtype=torch.float32)
        for at in range(0,len(self.names),4):
            if available_ram()<reserve:raise ValueError('RAM margin exhausted during cache construction; cache not published')
            batch=reader.cpu(self.names[at:at+4])
            if batch.dtype!=torch.float32 or batch.shape!=(min(4,len(self.names)-at),3,224,224) or not torch.isfinite(batch).all():
                raise ValueError('Expected finite original 224px FP32 CLIP tensors')
            values[at:at+len(batch)].copy_(batch)
        self._values=values;self.build_seconds=time.perf_counter()-start
    def reset_stats(self):self.stats={k:0. for k in ('decode_seconds','preprocess_seconds','lookup_seconds','transfer_seconds')}
    def cpu(self,names):
        start=time.perf_counter()
        try:indices=torch.tensor([self.index[n] for n in names],dtype=torch.long)
        except KeyError as error:raise ValueError('Image absent from this run pixel cache: '+str(error)) from None
        value=self._values.index_select(0,indices)  # copy: callers cannot mutate the bank
        if self.profile:self.stats['lookup_seconds']+=time.perf_counter()-start
        return value
    def __call__(self,names,device):
        value=self.cpu(names)
        if self.profile and torch.device(device).type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();result=value.to(device)
        if self.profile:
            if torch.device(device).type=='cuda':torch.cuda.synchronize()
            self.stats['transfer_seconds']+=time.perf_counter()-start
        return result
