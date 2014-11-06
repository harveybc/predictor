<?php
$this->breadcrumbs=array(
	'DocsEnLínea'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Documentos', 'url'=>array('index')),
	array('label' => 'Crear Doc. Online', 'url' => array('/Documentos/createOnline')),
	array('label'=>'Actualizar Documento', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Documento', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro de borrar esto?')),
	array('label'=>'Gestionar Documentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle('Detalles de Documento en Línea:<?php echo $model->descripcion; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		//'id',
		'usuario0.Username',
		'descripcion',
		'documento0.descripcion',
		'eliminado',
	),
)); ?>


