<?php
$this->breadcrumbs=array(
	'Aislamiento Tierra'=>array('index'),
	$model->Toma=>array('view','Toma'=>$model->Toma),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista mediciones', 'url'=>array('index')),
	array('label'=>'Nueva medida', 'url'=>array('create')),
	array('label'=>'Detalles detalles', 'url'=>array('view', 'id'=>$model->Toma)),
	array('label'=>'Gestionar mediciones', 'url'=>array('admin')),
);
?>

<?php
$nombre="";
$modelTMP=Motores::model()->findByAttributes(array('TAG'=>$model->TAG));
if (isset($modelTMP->Motor))
        $nombre=$modelTMP->Motor;
if (isset($modelTMP->TAG))
        $nombre=$nombre.' ('.$modelTMP->TAG.')';

?>

<?php $this->setPageTitle(' Actualizar medición Aislamiento Tierra de:'.$nombre.''); ?>
<div class="form"><style>    .forms50cr{            float:left;    }</style>


<?php 
$form=$this->beginWidget('CActiveForm', array(
	'id'=>'aislamiento-tierra-form',
	'enableAjaxValidation'=>true,
)); 

echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); 
?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
