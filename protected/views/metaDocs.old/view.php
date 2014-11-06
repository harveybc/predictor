<?php
$this->breadcrumbs=array(
	'documentos'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de documentos', 'url'=>array('index')),
	array('label'=>'Nuevo Metadocumento', 'url'=>array('create')),
	array('label'=>'Actualizar Metadocumento', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Metadocumento', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro de borrar esto??')),
	array('label'=>'Manage documentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle('Detalles de metadatos:<?php echo $model->titulo; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
         'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
	//	'id',
		'tipoContenido0.descripcion',
		'fabricante0.descripcion',
		'cerveceria0.descripcion',
		'numPedido',
		'numComision',
		'ubicacionT0.codigoSAP',
		'descripcion',
		'titulo',
		'version',
		'medio0.descripcion',
		'idioma0.descripcion',
		'disponibles',
		'existencias',
		'modulo',
		'columna',
		'fila',
		'documento0.descripcion',
		'ruta',
		'fechaCreacion',
		'fechaRecepcion',
		'autores',
		'usuario0.Username',
		'revisado',
		'userRevisado0.Username' ,
		'fechaRevisado',
		'eliminado',
		'secuencia0.descripcion',
		'ordenSecuencia0.posicion',
                'ISBN',
		'EAN13',
	),
)); ?>


<br /><h2 style="font-size:18px;color:#961C1F;">Tiene los siguientes préstamos :'); ?>
<ul><?php foreach($model->prestamoses as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->cedula, array('prestamos/view', 'id' => $foreignobj->id)));

				} ?>

</ul><br />

<h2 style="font-size:18px;color:#961C1F;">Tabla de Contenido de estos Metadatos:'); ?>
<ul><?php foreach($model->tablasDeContenidos as $foreignobj) { 

				printf('< li>%s</li>', CHtml::link($foreignobj->indice, array('tablasdecontenido/view', 'id' => $foreignobj->id)));

				} ?></ul>