<?php
$this->breadcrumbs=array(
	'Medios de Publicación'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Medios Publicación', 'url'=>array('index')),
	array('label'=>'Nuevo Medio Publicación', 'url'=>array('create')),
	array('label'=>'Actualizar Medio Publicación', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Medio Publicación', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro/a de borrar esto?')),
	array('label'=>'Gestionar Medios Publicación', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de Medios de Publicación #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'descripcion',
	),
)); ?>


<br /><?php $this->setPageTitle('Los siguientes documentos se encuentran en este medio: '); ?>
<ul><?php foreach($model->metaDocs as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->numPedido, array('metadocs/view', 'id' => $foreignobj->id)));

				} ?></ul>