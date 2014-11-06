<?php
$this->breadcrumbs=array(
	'Motores'=>array('index'),
	$model->id,
);
// busca el modelo del equipo en Mo
$modelTMP=Estructura::model()->findByAttributes(array('Equipo'=>$model->Equipo));

$this->menu=array(
	array('label'=>'Lista de Motores', 'url'=>array('index')),
	array('label'=>'Gestionar Motores', 'url'=>array('/motores/admin?id='.$modelTMP->id)),
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
    array_push($this->menu, array('label'=>'Nuevo Motor', 'url'=>array('/estructura/create?id=' . $model->Area)));
    array_push($this->menu,array('label'=>'Actualizar Motor', 'url'=>array('update', 'id'=>$model->id)));
}
if ($esAdmin) array_push($this->menu,array('label'=>'Borrar Motor', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro de borrar esto?')));

?>

<?php
$nombre="";
$cod="";
$modelTMP=Motores::model()->findByAttributes(array('TAG'=>$model->TAG));
if (isset($modelTMP->Motor))
        $nombre=$modelTMP->Motor;
if (isset($modelTMP->TAG))
        $nombre=$nombre.' ('.$modelTMP->TAG.')';
if (isset($modelTMP->Codigo))
        $cod=$cod.' '.$modelTMP->Codigo;

?>

<?php $this->setPageTitle('Detalles de Motor:<br/>'.$nombre.' (SAP:'.$cod.')'); ?>

<?php
$modelTM=Motores::model()->findByAttributes(array('TAG'=>$model->TAG));
?>
<div  class="forms50c">
<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		//'id',
		//'TAG',
                //'Motor',
                array(               // related city displayed as a link
            'label'=>'Motor',
            'type'=>'raw',
            'value'=>((isset($model->TAG))?$model->TAG:"")." - ".((isset($model->Motor))?$model->Motor:""),
        ),
                'Equipo',
                'Area',
		'Proceso',
		'kW',
		'Velocidad',
		'Marca',
		'Modelo',
		'Serie',
		'Rod_LC',
		'Rod_LA',
		'Lubricante',
		'IP',
		'Frame',
		'PathFoto',
                /*'plan_mant_vibraciones',
                'plan_mant_aislamiento',
                'plan_mant_lubricantes',
                'plan_mant_termografia'
                 * 
                 */
	),
)); ?>
</div>
<div  class="forms50c">
    <div  class="forms100cb">
    <?php 
if (is_numeric($model->PathFoto))
{
 echo '<a href="/index.php/archivos/displayArchivo?id='.$model->PathFoto.'">';
 echo '<img src="/index.php/archivos/displayArchivo?id='.$model->PathFoto.'" style="width:100%;border-width:1px;" />';
 echo '</a>';  
}
else
{
 echo '<a href="/index.php/reportes/passthru?path='.urlencode($model->PathFoto).'">';
 echo '<img src="/index.php/reportes/passthru?path='.urlencode($model->PathFoto).'" style="width:100%;border-width:1px;" />';
 echo '</a>';
}
   ?>
</div>
    </div>
