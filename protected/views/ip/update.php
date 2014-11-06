<?php
$this->breadcrumbs=array(
	'Ips'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Listaa de Ip', 'url'=>array('index')),
	array('label'=>'Nueva Ip', 'url'=>array('create')),
	array('label'=>'Ver Ip', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Ip', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Ip #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'ip-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
