<?php
$this->breadcrumbs=array(
	'Equipos'=>array('index'),
	$model->id,
);

// busca el modelo del equipo en estructura


$this->menu=array(
	array('label'=>'Lista de Equipos', 'url'=>array('index')),
	array('label'=>'Gestionar Equipos', 'url'=>array('/estructura/admin?proceso='.$model->Area)),
);
//TODO: provisional: para uso de roles de admin, ingeniero y usuario.
$esAdmin = 0;
$esIngeniero = 0;
if (!Yii::app()->user->isGuest) {
    $modeloU = Usuarios::model()->findBySql('select * from usuarios where Username="' . Yii::app()->user->name . '"');
}
if (isset($modeloU)) {
    $esAdmin = $modeloU->Es_administrador;
    $esIngeniero = $modeloU->Es_analista;
    if ($esAdmin)
        $esIngeniero = 1;
}
if ($esIngeniero)
{
    array_push($this->menu, array('label'=>'Nuevo Equipo', 'url'=>array('/estructura/create?id=' . $model->Area)));
    array_push($this->menu,array('label'=>'Actualizar Equipos', 'url'=>array('update', 'id'=>$model->id)));
}
if ($esAdmin) array_push($this->menu,array('label'=>'Borrar Equipo', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro de borrar esto?')));

	
	
	


?>

<?php
$nombre="";
$modelTMP=$model;
if (isset($modelTMP->Equipo))
        $nombre=$modelTMP->Equipo;
?>
<?php $this->setPageTitle ('Detalles de Equipo:<?php echo $nombre?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		//'id',
                'Codigo',
		'Equipo',
                'Area',
		'Proceso',
		'Indicativo',
                //'plan_mant_ultrasonido'
	),
)); ?>


