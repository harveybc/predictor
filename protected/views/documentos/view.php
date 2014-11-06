<?php
$this->breadcrumbs = array(
    'Documentos' => array('index'),
    $model->id,
);

$this->menu = array(
    array('label' => 'Lista de Documentos', 'url' => array('index')),
    array('label' => 'Nuevo Documento', 'url' => array('create')),
    array('label' => 'Actualizar Documento', 'url' => array('update', 'id' => $model->id)),
    array('label' => 'Borrar Documento', 'url' => '#', 'linkOptions' => array('submit' => array('delete', 'id' => $model->id), 'confirm' => 'Está seguro de borrar esto?')),
    array('label' => 'Manage Documentos', 'url' => array('admin')),
);
?>

<?php $this->setPageTitle('Detalles de Documento:<?php echo $model->descripcion; ?>'); ?>

<?php
$this->widget('zii.widgets.CDetailView', array(
    'data' => $model,
    'cssFile' => '/themes/detailview/styles.css',
    'attributes' => array(
        //'id',
        'descripcion',
        'permitirAdiciones',
        'permitirAnotaciones',
        'autorizarOtros',
        'requiereAutorizacion',
        'secuencia0.descripcion',
        'ordenSecuencia0.posicion',
        'eliminado',
        'conservacionInicio',
        'conservacionFin',
        'conservacionPermanente',
    ),
));
?>
<br/>
<h2 style="font-size:18px;color:#961C1F;"> Este documento tiene:'); ?>

<h2  style="font-size:16px;;"> Documentos: '); ?>
<ul><?php
foreach ($model->metaDocs as $foreignobj) {
    // Para mostrar los documentos electrónicos (Faltan los físicos y los online)
    if (isset($foreignobj->ruta))
        printf('<li>Documento electrónico: %s</li>', CHtml::link($foreignobj->ruta0->nombre, '/index.php/archivos/displayArchivo?id='.$foreignobj->ruta));
}
?></ul><br />

<br /><h2  style="font-size:16px;;"> Anotaciones: '); ?>
<ul><?php
foreach ($model->anotaciones as $foreignobj) {

    printf('<li>%s</li>', CHtml::link($foreignobj->descripcion, array('anotaciones/view', 'id' => $foreignobj->id)));
}
?></ul><br /><h2  style="font-size:16px;;"> Autorizaciones: '); ?>
<ul><?php
    foreach ($model->autorizaciones as $foreignobj) {

        printf('<li>%s</li>', CHtml::link($foreignobj->autorizado, array('autorizaciones/view', 'id' => $foreignobj->id)));
    }
?></ul><br /><h2  style="font-size:16px;;">Competencias: '); ?>
<ul><?php
    foreach ($model->competenciasDocumentoses as $foreignobj) {

        printf('<li>%s</li>', CHtml::link($foreignobj->id, array('competenciasDocumentos/view', 'id' => $foreignobj->id)));
    }
?></ul><br /><h2  style="font-size:16px;;"> Permisos: '); ?>
<ul><?php
    foreach ($model->permisoses as $foreignobj) {

        printf('<li>%s</li>', CHtml::link($foreignobj->id, array('permisos/view', 'id' => $foreignobj->id)));
    }
?></ul><br /><h2  style="font-size:16px;;"> Tags: '); ?>
<ul><?php
    foreach ($model->tags as $foreignobj) {

        printf('<li>%s</li>', CHtml::link($foreignobj->descripcion, array('tags/view', 'id' => $foreignobj->id)));
    }
?></ul>