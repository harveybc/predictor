<?php
$this->breadcrumbs=array(
	'Avisos Zis'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Avisos ZI', 'url'=>array('index')),
	//array('label'=>'Crear Aviso ZI', 'url'=>array('create')),
	//array('label'=>'Actualizar Aviso ZI', 'url'=>array('update', 'id'=>$model->id)),
	//array('label'=>'Borrar Aviso ZI', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Are you sure you want to delete this item?')),
	array('label'=>'Gestionar Avisos ZI', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de aviso ZI para plan de mantenimiento #<?php echo $model->plan_mant; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile' => '/themes/detailview/styles.css',
	'attributes'=>array(
		'Ruta',
                'Codigo',
		'Operador',
		'Fecha',
		'Estado',
		'Observaciones',
		'arreglado',
		'plan_mant',
		'OT',
	),
)); ?>


