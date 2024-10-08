import torch,random

def _batch_by_token_count(idx_target_lengths, max_frames, batch_size=None):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > max_frames or (batch_size and len(current_batch) == batch_size):
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches

class BatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, lengths,max_frames,num_buckets=50,shuffle=False):
        
        max_length = max(lengths)
        min_length = min(lengths)

        assert max_frames >= max_length
        self.max_frames=max_frames

        
        idx_lengths=[(idx,l) for idx,l in enumerate(lengths)]
            
        if shuffle==True:
            random.shuffle(idx_lengths)
            
            
        self.batches=[]
        current_batch=[]
        
        current_token_idx=0
        
        cmax_l=0
        for i,(idx,l) in enumerate(idx_lengths):
            cmax_l=max(cmax_l,l)
            if current_token_idx < i - max_frames//cmax_l + 1:
                cmax_l=l
                self.batches.append(current_batch)
                current_batch=[idx]
                current_token_idx=i
            else:
                current_batch.append(idx)
                
        if current_batch:
            self.batches.append(current_batch)
        
        if shuffle:
            random.shuffle(self.batches)
        
#         current_token_count=0
#         for idx,tgt_length in idx_lengths:
#             if current_token_count+tgt_length>max_frames:
#                 self.batches.append(current_batch)
#                 current_batch=[idx]
#                 current_token_count=tgt_length
#             else:
#                 current_batch.append(idx)
#                 current_token_count+=tgt_length
#         if current_batch:
#             self.batches.append(current_batch)

    def __iter__(self):
        
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)



class CustomBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, lengths,max_frames,num_buckets,shuffle=False):
        
        max_length = max(lengths)
        min_length = min(lengths)

        assert max_frames >= max_length
        self.max_frames=max_frames

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths = torch.tensor(lengths)
        bucket_assignments = torch.bucketize(lengths, buckets)

#         self.
        idx_length_buckets = [(idx, length, bucket_assignments[idx]) for idx, length in enumerate(lengths)]
        self.shuffle=shuffle
        if shuffle:
            idx_length_buckets = random.sample(idx_length_buckets, len(idx_length_buckets))
        else:
            idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[1], reverse=True)

        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            max_frames,
            batch_size=None,
        )
        
        if shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
#         if self.shuffle:
#             idx_length_buckets = random.sample(self.idx_length_buckets, len(self.idx_length_buckets))
#         else:
#             idx_length_buckets = sorted(self.idx_length_buckets, key=lambda x: x[1], reverse=True)

#         sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
#         batches = _batch_by_token_count(
#             [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
#             self.max_frames,
#             batch_size=None,
#         )
        
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        lengths,
        max_frames,
        num_buckets,
        shuffle=False,
        batch_size=None,
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        assert max_frames >= max_length

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths = torch.tensor(lengths)
        bucket_assignments = torch.bucketize(lengths, buckets)

        idx_length_buckets = [(idx, length, bucket_assignments[idx]) for idx, length in enumerate(lengths)]
        if shuffle:
            idx_length_buckets = random.sample(idx_length_buckets, len(idx_length_buckets))
        else:
            idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[1], reverse=True)

        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            max_frames,
            batch_size=batch_size,
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_fn):
        self.dataset = dataset
        self.transform_fn = transform_fn

    def __getitem__(self, idx):
        return self.transform_fn(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)