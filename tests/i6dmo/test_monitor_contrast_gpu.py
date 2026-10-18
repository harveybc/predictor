"""Explicit small synthetic runtime checks; never a scientific measurement."""
import numpy as np
import pytest


def model_config():
    from tools.i6dmo import continuation_models as M
    return M.build_config(['a', 'b'], 13, 2, [2,3,4,5,6,12,18], head='lstm',core=[2])


def inputs(n=12):
    rng=np.random.default_rng(712)
    return [rng.normal(size=(n,13,2)).astype('float32'),
            np.tile(np.arange(-24,1,2,dtype='float32'),(n,1)),np.zeros((n,13),dtype='float32')]


def test_bn_preserves_existing_weights_and_inference_statistics(tmp_path):
    from tools.i6dmo import continuation_models as M, monitor_contrast as MC
    original=M.build(model_config())
    bundle,proof=MC.insert_bn(original.forecast_model,original.config)
    x=inputs()
    for layer in original.forecast_model.layers:
        for a,b in zip(layer.get_weights(),bundle.forecast_model.get_layer(layer.name).get_weights(),strict=True):
            np.testing.assert_array_equal(a,b)
    bn=bundle.forecast_model.get_layer('core_batch_normalization')
    before=bn.get_weights()
    p=bundle.forecast_model(x,training=False)
    for a,b in zip(before,bn.get_weights(),strict=True):np.testing.assert_array_equal(a,b)
    for a,b in zip(p,original.forecast_model(x,training=False),strict=True):
        np.testing.assert_allclose(a,b,atol=1e-7,rtol=1e-6)
    path=tmp_path/'model.keras';bundle.forecast_model.save(path)
    restored=M.keras.models.load_model(path,compile=False,safe_mode=True)
    for a,b in zip(restored(x,training=False),p,strict=True):np.testing.assert_array_equal(a,b)
    assert proof['new_layer']=='core_batch_normalization'


def test_real_trainer_same_gradients_and_explicit_relative_monitor():
    from tools.i6dmo import continuation_models as M, train as T, monitor_contrast as MC
    x=inputs();y=np.ones((12,7));mask=np.ones((12,7),bool)
    fit=(x,y,mask);inner=([a[:4] for a in x],y[:4]*2,mask[:4])
    settings=dict(batch_size=4,learning_rate=.001,weight_decay=.0001,patience=10,min_delta=1e-8,max_epochs=2)
    first=T.fit_dmo(M.build(model_config()),fit,inner,settings,6102026)
    adapter=MC.MonitorAdapter(np.arange(12),[1.]*7,[2.]*7,'legacy4')
    second=T.fit_dmo(M.build(model_config()),fit,inner,settings,6102026,monitor_adapter=adapter)
    assert first['last_weights_sha256']==second['last_weights_sha256']
    assert first['restored_weights_sha256']==second['restored_weights_sha256']
    adapter=MC.MonitorAdapter(np.arange(6,12),[1.]*7,[2.]*7,'relative1')
    third=T.fit_dmo(M.build(model_config()),fit,inner,settings,6102026,monitor_adapter=adapter)
    assert third['last_weights_sha256']==first['last_weights_sha256']
    assert third['updates']==6
    for row in third['history']:
        assert row['monitored']==pytest.approx(.5*(row['fit']['loss']+row['inner']['loss']))
    assert third['monitor_evaluation']['self_check']=='PASS'


def test_bn_real_training_and_monitor_reload():
    from tools.i6dmo import continuation_models as M, train as T, monitor_contrast as MC
    x=inputs();y=np.ones((12,7));mask=np.ones((12,7),bool)
    fit=(x,y,mask);inner=([a[:4] for a in x],y[:4]*2,mask[:4])
    bundle,_=MC.insert_bn(M.build(model_config()).forecast_model,model_config())
    adapter=MC.MonitorAdapter(np.arange(12),[1.]*7,[2.]*7,'relative4_bn')
    r=T.fit_dmo(bundle,fit,inner,dict(batch_size=4,learning_rate=.001,weight_decay=.0001,
        patience=10,min_delta=1e-8,max_epochs=2),6102026,monitor_adapter=adapter)
    assert r['monitor_evaluation']['self_check']=='PASS'
