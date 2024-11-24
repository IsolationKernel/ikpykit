# ikdc

拼写错误，应为idkc

```
(CDiDS) PS D:\MYX\isoml> python .\isoml\cluster\tests\test_ikdc.py
Traceback (most recent call last):
  File ".\isoml\cluster\tests\test_ikdc.py", line 75, in <module>
    test_IKDC()
  File ".\isoml\cluster\tests\test_ikdc.py", line 56, in test_IKDC
    labels_pred = ikdc.fit_predict(X)
  File "C:\Users\Admin\anaconda3\envs\CDiDS\lib\site-packages\sklearn\base.py", line 791, in fit_predict
    self.fit(X)
  File "D:\MYX\isoml\isoml\cluster\_ikdc.py", line 122, in fit
    self._fit(data_ik)
  File "D:\MYX\isoml\isoml\cluster\_ikdc.py", line 132, in _fit
    init_center = self._initialize_centers(X, data_index)
  File "D:\MYX\isoml\isoml\cluster\_ikdc.py", line 166, in _initialize_centers
    seeds_id = self._get_seeds(X[samples_index])
  File "D:\MYX\isoml\isoml\cluster\_ikdc.py", line 200, in _get_seeds
    sort_mult = np.argpartition(mult, -self.k, axis=1)[-self.k :]
  File "<__array_function__ internals>", line 180, in argpartition
  File "C:\Users\Admin\anaconda3\envs\CDiDS\lib\site-packages\numpy\core\fromnumeric.py", line 845, in argpartition
    return _wrapfunc(a, 'argpartition', kth, axis=axis, kind=kind, order=order)
  File "C:\Users\Admin\anaconda3\envs\CDiDS\lib\site-packages\numpy\core\fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
numpy.AxisError: axis 1 is out of bounds for array of dimension 1
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

# ikgod

(CDiDS) PS D:\MYX\isoml> python .\isoml\graph\tests\test_ikgod.py

可以运行

# ikgad

(CDiDS) PS D:\MYX\isoml> python .\isoml\group\tests\test_ikgad.py

可以运行

# streaKHC

暂无HX版本测试

# ikast

# ikat

