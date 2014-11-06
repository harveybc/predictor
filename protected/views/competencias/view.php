<?php
$this->breadcrumbs=array(
	'Competencias'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Competencias', 'url'=>array('index')),
	array('label'=>'Nueva Competencias', 'url'=>array('create')),
	array('label'=>'Actualizar Competencias', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Competencias', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro/a de borrar esto?')),
	array('label'=>'Gestionar Competencias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de Competencias #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'descripcion',
	),
)); ?>


<br /><?php $this->setPageTitle(' Las siguientes Competencias de Documentospertenecen a estas Competencias: '); ?>
<ul><?php foreach($model->competenciasDocumentoses as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->id, array('competenciasdocumentos/view', 'id' => $foreignobj->id)));

				} ?></ul><br /><?php $this->setPageTitle(' Las siguientes Competencias de Usuarios pertenecen a estas Competencias: '); ?>
<ul><?php foreach($model->competenciasUsuarioses as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->id, array('competenciasusuarios/view', 'id' => $foreignobj->id)));

				} ?></ul>