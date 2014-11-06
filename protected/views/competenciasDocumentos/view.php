<?php
$this->breadcrumbs=array(
	'Competencias Documentos'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de CompetenciasDocumentos', 'url'=>array('index')),
	array('label'=>'Create CompetenciasDocumentos', 'url'=>array('create')),
	array('label'=>'Update CompetenciasDocumentos', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Delete CompetenciasDocumentos', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Are you sure you want to delete this item?')),
	array('label'=>'Manage CompetenciasDocumentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('View CompetenciasDocumentos #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
	'attributes'=>array(
		'id',
		'documento0.descripcion',
		'competencia0.descripcion',
	),
)); ?>


