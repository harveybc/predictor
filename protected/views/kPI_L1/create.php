<?php
$this->breadcrumbs=array(
	'Kpi  L1s'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Create'),
);

$this->menu=array(
	array('label'=>'List KPI_L1', 'url'=>array('index')),
	array('label'=>'Manage KPI_L1', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Create KPI_L1 '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'kpi--l1-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Create')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
