<?php
$this->breadcrumbs=array(
	'Competencias Usuarioses'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de CompetenciasUsuarios', 'url'=>array('index')),
	array('label'=>'Create CompetenciasUsuarios', 'url'=>array('create')),
	array('label'=>'Update CompetenciasUsuarios', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Delete CompetenciasUsuarios', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Are you sure you want to delete this item?')),
	array('label'=>'Manage CompetenciasUsuarios', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('View CompetenciasUsuarios #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
	'attributes'=>array(
		'id',
		'usuario0.Username',
		'competencia0.descripcion',
	),
)); ?>


