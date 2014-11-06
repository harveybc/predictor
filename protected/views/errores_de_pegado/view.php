<?php
$this->breadcrumbs=array(
	'Errores De Pegados'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'List Errores_de_pegado', 'url'=>array('index')),
	array('label'=>'Create Errores_de_pegado', 'url'=>array('create')),
	array('label'=>'Update Errores_de_pegado', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Delete Errores_de_pegado', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro de borrar esto?')),
	array('label'=>'Manage Errores_de_pegado', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('View Errores_de_pegado #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'Campo0',
		'Campo1',
		'Campo2',
		'Campo3',
		'Campo4',
		'Campo5',
		'Campo6',
		'Campo7',
		'Campo8',
		'Campo9',
		'Campo10',
		'Campo11',
		'Campo12',
		'Campo13',
		'Campo14',
	),
)); ?>


