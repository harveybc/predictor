<?php

class Termotablero extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'termotablero';
	}

	public function rules()
	{
		return array(
			array('OT, Tamano', 'numerical', 'integerOnly'=>true),
			array('TAG, Analista, Criterio', 'length', 'max'=>50),
                        array('Estado', 'numerical', 'integerOnly'=>true),
                        array('Observaciones', 'length', 'max'=>250),
			array('Path', 'length', 'max'=>200),
			array('Fecha', 'safe'),
			array(' OT, Estado, Observaciones,id, Fecha, OT, TAG, Path, Analista, Tamano, Criterio', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
		);
	}

	public function behaviors()
	{
		return array('CAdvancedArBehavior',
				array('class' => 'ext.CAdvancedArBehavior')
				);
	}

	public function attributeLabels()
	{
		return array(
			'id' => Yii::t('app', 'ID'),
			'Fecha' => Yii::t('app', 'Fecha'),
			'OT' => Yii::t('app', 'Orden de Trabajo'),
			'TAG' => Yii::t('app', 'Tag'),
			'Path' => Yii::t('app', 'Path'),
			'Analista' => Yii::t('app', 'Analista'),
			'Tamano' => Yii::t('app', 'Tamano'),
			'Criterio' => Yii::t('app', 'Criterio'),
                        'Estado' => Yii::t('app', 'Estado'),
                        'Observaciones' => Yii::t('app', 'Observaciones'),            
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;
		$criteria->compare('id',$this->id,true);
		$criteria->compare('Fecha',$this->Fecha,true);
		$criteria->compare('OT',$this->OT);
		$criteria->compare('TAG',$this->TAG,true);

                $criteria->compare('Estado', $this->Estado);
		//$criteria->compare('Path',$this->Path,true);
		$criteria->compare('Analista',$this->Analista,true);
//		$criteria->compare('Tamano',$this->Tamano);
//		$criteria->compare('Criterio',$this->Criterio,true);
		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
        
         public function searchTableros($TAG)
        {
            if ($TAG=="")
                $TAG="ZZ_ZZ_Z";
                        $criteria = new CDbCriteria;
                        
                        $criteria->compare('id',$this->id,true);
            $criteria->compare('TAG',
                               $TAG,
                               false);
            return new CActiveDataProvider(get_class($this), array (
                'criteria' => $criteria,
            ));
        }

}
