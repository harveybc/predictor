<?php

class Tableros extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'tableros';
	}

	public function rules()
	{
		return array(
                        array ('TAG,Tablero','required'),
                        array('plan_mant_termografia', 'numerical', 'integerOnly'=>true),
			array('Proceso, Area, TAG, Tablero', 'length', 'max'=>50),
			array('id, Proceso, Area, TAG, Tablero', 'safe', 'on'=>'search'),
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
			'Proceso' => Yii::t('app', 'Area'),
			'Area' => Yii::t('app', 'Proceso'),
			'TAG' => Yii::t('app', 'TAG'),
			'Tablero' => Yii::t('app', 'Nombre del tablero'),
                        'plan_mant_termografia' => Yii::t('app', 'Plan de mantenimiento de termografía de tablero'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('Proceso',$this->Proceso,true);

		$criteria->compare('Area',$this->Area,true);

		$criteria->compare('TAG',$this->TAG,true);

		$criteria->compare('Tablero',$this->Tablero,true);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
        
        public function searchTableros($area)
        {
            if ($area=="")
                $area="ZZ_ZZ_Z";
                        $criteria = new CDbCriteria;
            $criteria->compare('Area',
                               $area,
                               true);
            return new CActiveDataProvider(get_class($this), array (
                'criteria' => $criteria,
            ));
            /*
            $sql = 'SELECT * FROM motores WHERE Equipo="' . $equipo . '" AND Area="'.$area.'" ORDER BY Equipo';
            $myDataProvider = new CSqlDataProvider($sql, array (
                    'keyField' => 'id',
                    'pagination' => array (
                        'pageSize' => 10,
                    ),
                    )
            );
            return $myDataProvider;
             * 
             */
        }
}
