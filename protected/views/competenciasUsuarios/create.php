<?php
$this->breadcrumbs=array(
	'Competencias Usuarioses'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Create'),
);

$this->menu=array(
	array('label'=>'Lista de CompetenciasUsuarios', 'url'=>array('index')),
	array('label'=>'Manage CompetenciasUsuarios', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Create CompetenciasUsuarios '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'competencias-usuarios-form',
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
