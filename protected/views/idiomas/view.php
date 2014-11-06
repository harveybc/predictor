<?php
$this->breadcrumbs=array(
	'Idiomas'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Idiomas', 'url'=>array('index')),
	array('label'=>'Nuevo Idiomas', 'url'=>array('create')),
	array('label'=>'Actualizar Idiomas', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Idiomas', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro/a de borrar esto?')),
	array('label'=>'Gestionar Idiomas', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de Idiomas #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'descripcion',
	),
)); ?>


<br /><?php $this->setPageTitle(' Los siguientes documentos se encuentran en este idioma'); ?>
<ul><?php foreach($model->metaDocs as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->numPedido, array('metadocs/view', 'id' => $foreignobj->id)));

				} ?></ul>