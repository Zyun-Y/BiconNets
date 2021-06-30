import torch

class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()


    def preload(self):
        try:
            self.next_input, self.next_target,self.next_conn = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_conn  = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_conn= self.next_conn.cuda(non_blocking=True)
            self.next_input = self.next_input.float() #if need
            self.next_target = self.next_target.float() #if need
            self.next_conn = self.next_conn.float() #if need

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        conn = self.next_conn
        self.preload()
        return input, target,conn
