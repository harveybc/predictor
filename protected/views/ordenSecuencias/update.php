<?php
$this->breadcrumbs=array(
	'Orden Secuenciases'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Update'),
);

$this->menu=array(
	array('label'=>'Lista de OrdenSecuencias', 'url'=>array('index')),
	array('label'=>'Create OrdenSecuencias', 'url'=>array('create')),
	array('label'=>'View OrdenSecuencias', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Manage OrdenSecuencias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Update OrdenSecuencias #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'orden-secuencias-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Update')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
