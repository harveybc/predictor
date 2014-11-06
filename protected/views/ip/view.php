<?php
$this->breadcrumbs=array(
	'Ips'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Ip', 'url'=>array('index')),
	array('label'=>'Nueva Ip', 'url'=>array('create')),
	array('label'=>'Actualizar Ip', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Ip', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro de borrar esto?')),
	array('label'=>'Gestionar Ip', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles Ip #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'Fecha',
		'TAG',
		'IP',
	),
)); ?>


