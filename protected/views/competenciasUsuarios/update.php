<?php
$this->breadcrumbs=array(
	'Competencias Usuarioses'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Update'),
);

$this->menu=array(
	array('label'=>'Lista de CompetenciasUsuarios', 'url'=>array('index')),
	array('label'=>'Create CompetenciasUsuarios', 'url'=>array('create')),
	array('label'=>'View CompetenciasUsuarios', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Manage CompetenciasUsuarios', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Update CompetenciasUsuarios #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'competencias-usuarios-form',
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
