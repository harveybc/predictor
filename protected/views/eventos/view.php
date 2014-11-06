<?php
$this->breadcrumbs=array(
	'Eventos'=>array('index'),
	$model->id,
);

$this->menu=array(
	//array('label'=>'Lista de Eventos', 'url'=>array('index')),
	
	array('label'=>'Gestionar Eventos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('View Eventos #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
	'attributes'=>array(
		'id',
		'usuario',
		'modulo',
		'operacion',
		'ip',
		'descripcion',
		'fecha',
	),
)); ?>


