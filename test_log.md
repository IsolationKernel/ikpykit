# ikdc

拼写错误，应为idkc

```
(CDiDS) PS D:\MYX\isoml> python .\myx_test\ikdc\test.py
Traceback (most recent call last):
  File ".\myx_test\ikdc\test.py", line 35, in <module>
    predict = ikdc.fit_predict(data)
  File "C:\Users\Admin\anaconda3\envs\CDiDS\lib\site-packages\sklearn\base.py", line 791, in fit_predict
    self.fit(X)
  File "D:\MYX\isoml\isoml\cluster\_ikdc.py", line 123, in fit
    self._fit(data_ik)
  File "D:\MYX\isoml\isoml\cluster\_ikdc.py", line 140, in _fit
    c_mean = sp.vstack([c.kernel_mean for c in self.clusters_])
  File "C:\Users\Admin\anaconda3\envs\CDiDS\lib\site-packages\scipy\sparse\_construct.py", line 569, in vstack
    return bmat([[b] for b in blocks], format=format, dtype=dtype)
  File "C:\Users\Admin\anaconda3\envs\CDiDS\lib\site-packages\scipy\sparse\_construct.py", line 618, in bmat
    raise ValueError('blocks must be 2-D')
ValueError: blocks must be 2-D
```

# pskc

```
(CDiDS) PS D:\MYX\isoml> python .\isoml\cluster\tests\test_pskc.py
Traceback (most recent call last):
  File ".\isoml\cluster\tests\test_pskc.py", line 42, in <module>
    test_PSKC()
  File ".\isoml\cluster\tests\test_pskc.py", line 36, in test_PSKC
    labels_pred = clus.fit_predict(X)
  File "C:\Users\Admin\anaconda3\envs\CDiDS\lib\site-packages\sklearn\base.py", line 791, in fit_predict
    self.fit(X)
  File "D:\MYX\isoml\isoml\cluster\_pskc.py", line 107, in fit
    self._fit(ndata)
  File "D:\MYX\isoml\isoml\cluster\_pskc.py", line 121, in _fit
    c_k, point_indices = self._update_cluster(
  File "D:\MYX\isoml\isoml\cluster\_pskc.py", line 169, in _update_cluster
    assert self._get_n_points() == X.shape[0] - len(point_indices)
AssertionError
```

# ikhc

```
(CDiDS) PS D:\MYX\isoml> python .\isoml\cluster\tests\test_ikhc.py
D:\MYX\isoml\isoml\cluster\_ikhc.py:88: ClusterWarning: scipy.cluster: The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix
  self.dendrogram_ = linkage(1 - similarity_matrix, method=self.lk_method)
Traceback (most recent call last):
  File ".\isoml\cluster\tests\test_ikhc.py", line 53, in <module>
    test_IsoKHC()
  File ".\isoml\cluster\tests\test_ikhc.py", line 47, in test_IsoKHC
    labels_pred = clus.fit_predict(X)
  File "C:\Users\Admin\anaconda3\envs\CDiDS\lib\site-packages\sklearn\base.py", line 792, in fit_predict
    return self.labels_
AttributeError: 'IsoKHC' object has no attribute 'labels_'
```

# ikgod

原文中有lambda参数，但该版本没有？

```
(CDiDS) PS D:\MYX\isoml> python .\myx_test\ikgod\test.py
Traceback (most recent call last):
  File ".\myx_test\ikgod\test.py", line 32, in <module>
    ikgod.fit(adj, attr, para_dict["h"])
  File "D:\MYX\isoml\isoml\graph\_ikgod.py", line 133, in fit
    self._fit(adjacency, features, h)
  File "D:\MYX\isoml\isoml\graph\_ikgod.py", line 164, in _fit
    self.embedding_ = self._subgraph_embeddings(adjacency, features_trans, h_index)
  File "D:\MYX\isoml\isoml\graph\_ikgod.py", line 186, in _subgraph_embeddings
    subgraph_feat = subgraph_feat - np.tile(
  File "C:\Users\Admin\anaconda3\envs\CDiDS\lib\site-packages\scipy\sparse\_base.py", line 494, in __sub__
    return self._sub_dense(other)
  File "C:\Users\Admin\anaconda3\envs\CDiDS\lib\site-packages\scipy\sparse\_base.py", line 456, in _sub_dense
    return self.todense() - other
  File "C:\Users\Admin\anaconda3\envs\CDiDS\lib\site-packages\scipy\sparse\_base.py", line 502, in __rsub__
    raise NotImplementedError('subtracting a sparse matrix from a '
NotImplementedError: subtracting a sparse matrix from a nonzero scalar is not supported```

# ikgad

(CDiDS) PS D:\MYX\isoml> python .\isoml\group\tests\test_ikgad.py

可以运行

# streaKHC

暂无HX版本测试

# ikast

# ikat

# iforest

未实现